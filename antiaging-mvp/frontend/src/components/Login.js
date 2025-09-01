import React from 'react';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import { useContext } from 'react';
import { AuthContext } from '../contexts/AuthContext';

const schema = yup.object({
  username: yup.string().required(),
  password: yup.string().required(),
}).required();

const Login = () => {
  const { register, handleSubmit, formState: { errors } } = useForm({ resolver: yupResolver(schema) });
  const { login } = useContext(AuthContext);

  const onSubmit = data => login(data.username, data.password);

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input {...register('username')} placeholder="Username" />
      <p>{errors.username?.message}</p>
      <input {...register('password')} type="password" placeholder="Password" />
      <p>{errors.password?.message}</p>
      <button type="submit">Login</button>
    </form>
  );
};

export default Login;